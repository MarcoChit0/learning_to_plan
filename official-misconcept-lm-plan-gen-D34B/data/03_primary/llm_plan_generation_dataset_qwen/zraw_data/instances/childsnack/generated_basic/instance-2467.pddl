; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 388947

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child224 child179 - child
    bread260 bread432 - bread-portion
    content136 content451 - content-portion
    tray425 tray61 - tray
    table171 table40 table326 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray425 kitchen)
     (at tray61 kitchen)
     (at_kitchen_bread bread260)
     (at_kitchen_bread bread432)
     (at_kitchen_content content136)
     (at_kitchen_content content451)
     (not_allergic_gluten child179)
     (not_allergic_gluten child224)
     (waiting child224 table40)
     (waiting child179 table326)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child224)
     (served child179)
    )
  )
)
