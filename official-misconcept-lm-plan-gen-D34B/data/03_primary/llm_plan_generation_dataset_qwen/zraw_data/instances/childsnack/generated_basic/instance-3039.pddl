; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 171067

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child337 child498 - child
    bread16 bread318 - bread-portion
    content248 content173 - content-portion
    tray393 tray354 tray21 - tray
    table240 table149 table99 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray393 kitchen)
     (at tray354 kitchen)
     (at tray21 kitchen)
     (at_kitchen_bread bread16)
     (at_kitchen_bread bread318)
     (at_kitchen_content content248)
     (at_kitchen_content content173)
     (no_gluten_bread bread318)
     (no_gluten_content content248)
     (allergic_gluten child337)
     (not_allergic_gluten child498)
     (waiting child337 table99)
     (waiting child498 table99)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child337)
     (served child498)
    )
  )
)
