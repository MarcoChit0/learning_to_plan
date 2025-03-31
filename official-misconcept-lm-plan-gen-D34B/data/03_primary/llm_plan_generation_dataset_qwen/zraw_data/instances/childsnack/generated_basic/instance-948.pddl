; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 660870

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child65 child491 child152 - child
    bread117 bread129 bread359 - bread-portion
    content247 content282 content276 - content-portion
    tray220 tray8 tray254 tray463 - tray
    table137 table24 table486 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray220 kitchen)
     (at tray8 kitchen)
     (at tray254 kitchen)
     (at tray463 kitchen)
     (at_kitchen_bread bread117)
     (at_kitchen_bread bread129)
     (at_kitchen_bread bread359)
     (at_kitchen_content content247)
     (at_kitchen_content content282)
     (at_kitchen_content content276)
     (not_allergic_gluten child65)
     (not_allergic_gluten child491)
     (not_allergic_gluten child152)
     (waiting child65 table137)
     (waiting child491 table137)
     (waiting child152 table137)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child65)
     (served child491)
     (served child152)
    )
  )
)
