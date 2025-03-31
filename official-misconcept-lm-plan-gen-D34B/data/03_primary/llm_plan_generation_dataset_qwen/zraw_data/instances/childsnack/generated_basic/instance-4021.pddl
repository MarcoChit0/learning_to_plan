; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 99666

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child244 child14 child113 - child
    bread291 bread412 bread408 - bread-portion
    content39 content111 content50 - content-portion
    tray471 - tray
    table497 table338 table97 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray471 kitchen)
     (at_kitchen_bread bread291)
     (at_kitchen_bread bread412)
     (at_kitchen_bread bread408)
     (at_kitchen_content content39)
     (at_kitchen_content content111)
     (at_kitchen_content content50)
     (not_allergic_gluten child14)
     (not_allergic_gluten child244)
     (not_allergic_gluten child113)
     (waiting child244 table497)
     (waiting child14 table97)
     (waiting child113 table338)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child244)
     (served child14)
     (served child113)
    )
  )
)
