# TCP

## TCP Packet Format
 
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 +-------------------------------------------------------------------------------------+
 |     Source Port (16 bits)    |  Destination Port (16 bits)                      |
 +-------------------------------------------------------------------------------------+
 |                Sequence Number (32 bits)                                          |
 +-------------------------------------------------------------------------------------+
 |            Acknowledgment Number (32 bits)                                        |
 +-------------------------------------------------------------------------------------+
 | Data  |Rese-|  Control Flags  |   Window Size (16 bits)   |   Checksum (16 bits)   |
 |Offset |rved |  (6 bits)       |                           |                       |
 +-------------------------------------------------------------------------------------+
 |           Urgent Pointer (16 bits)                                                 |
 +-------------------------------------------------------------------------------------+
 |                Options (variable length)                                           |
 +-------------------------------------------------------------------------------------+
 |                               Data (variable length)                                |
 +-------------------------------------------------------------------------------------+
 
 Note: Each field is represented in bits, and the total length of a TCP packet can vary.
 Start with a minimum of 20 bytes (without options).
 
 The control flags include:
 - URG: Urgent Pointer field significant
 - ACK: Acknowledgment field significant
 - PSH: Push Function
 - RST: Reset the connection
 - SYN: Synchronize sequence numbers
 - FIN: No more data from the sender
 
 The window size indicates the size of the sender's receive window (or buffer).
 
 The checksum is used for error-checking the header and data.
 
 The options field can include various TCP options, such as Maximum Segment Size (MSS).
 
 The data field contains the actual data being transmitted.
 
 This ASCII art representation provides a visual understanding of the TCP packet structure.
 
