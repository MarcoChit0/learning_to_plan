; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 503194

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child115 - child
    bread444 - bread-portion
    content371 - content-portion
    tray77 tray324 tray303 - tray
    table320 table287 table469 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray77 kitchen)
     (at tray324 kitchen)
     (at tray303 kitchen)
     (at_kitchen_bread bread444)
     (at_kitchen_content content371)
     (not_allergic_gluten child115)
     (waiting child115 table320)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child115)
    )
  )
)
