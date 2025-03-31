; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 100566

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child422 child380 - child
    bread85 bread332 - bread-portion
    content394 content115 - content-portion
    tray54 tray295 - tray
    table232 table335 table70 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray54 kitchen)
     (at tray295 kitchen)
     (at_kitchen_bread bread85)
     (at_kitchen_bread bread332)
     (at_kitchen_content content394)
     (at_kitchen_content content115)
     (not_allergic_gluten child422)
     (not_allergic_gluten child380)
     (waiting child422 table335)
     (waiting child380 table335)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child422)
     (served child380)
    )
  )
)
