; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 428343

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child31 child174 child478 - child
    bread414 bread4 bread63 - bread-portion
    content351 content483 content230 - content-portion
    tray144 - tray
    table385 table423 table400 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray144 kitchen)
     (at_kitchen_bread bread414)
     (at_kitchen_bread bread4)
     (at_kitchen_bread bread63)
     (at_kitchen_content content351)
     (at_kitchen_content content483)
     (at_kitchen_content content230)
     (not_allergic_gluten child478)
     (not_allergic_gluten child31)
     (not_allergic_gluten child174)
     (waiting child31 table423)
     (waiting child174 table385)
     (waiting child478 table423)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child31)
     (served child174)
     (served child478)
    )
  )
)
