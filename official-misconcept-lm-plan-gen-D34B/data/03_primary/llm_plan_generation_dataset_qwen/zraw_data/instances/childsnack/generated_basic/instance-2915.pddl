; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 261162

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child270 child85 - child
    bread464 bread267 - bread-portion
    content118 content119 - content-portion
    tray245 tray270 tray285 - tray
    table69 table479 table13 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray245 kitchen)
     (at tray270 kitchen)
     (at tray285 kitchen)
     (at_kitchen_bread bread464)
     (at_kitchen_bread bread267)
     (at_kitchen_content content118)
     (at_kitchen_content content119)
     (no_gluten_bread bread267)
     (no_gluten_content content119)
     (allergic_gluten child85)
     (not_allergic_gluten child270)
     (waiting child270 table479)
     (waiting child85 table13)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child270)
     (served child85)
    )
  )
)
