; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 630993

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child92 child148 - child
    bread226 bread364 - bread-portion
    content459 content300 - content-portion
    tray442 tray356 - tray
    table83 table488 table143 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray442 kitchen)
     (at tray356 kitchen)
     (at_kitchen_bread bread226)
     (at_kitchen_bread bread364)
     (at_kitchen_content content459)
     (at_kitchen_content content300)
     (no_gluten_bread bread364)
     (no_gluten_content content459)
     (allergic_gluten child148)
     (not_allergic_gluten child92)
     (waiting child92 table143)
     (waiting child148 table143)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child92)
     (served child148)
    )
  )
)
