; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 642983

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child277 child498 - child
    bread290 bread152 - bread-portion
    content88 content62 - content-portion
    tray462 tray389 tray486 - tray
    table97 table120 table406 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray462 kitchen)
     (at tray389 kitchen)
     (at tray486 kitchen)
     (at_kitchen_bread bread290)
     (at_kitchen_bread bread152)
     (at_kitchen_content content88)
     (at_kitchen_content content62)
     (not_allergic_gluten child498)
     (not_allergic_gluten child277)
     (waiting child277 table97)
     (waiting child498 table120)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child277)
     (served child498)
    )
  )
)
