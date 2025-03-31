; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 253162

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child136 child360 - child
    bread472 bread197 - bread-portion
    content346 content219 - content-portion
    tray482 tray13 tray216 tray94 - tray
    table309 table22 table481 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray482 kitchen)
     (at tray13 kitchen)
     (at tray216 kitchen)
     (at tray94 kitchen)
     (at_kitchen_bread bread472)
     (at_kitchen_bread bread197)
     (at_kitchen_content content346)
     (at_kitchen_content content219)
     (not_allergic_gluten child136)
     (not_allergic_gluten child360)
     (waiting child136 table309)
     (waiting child360 table22)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child136)
     (served child360)
    )
  )
)
