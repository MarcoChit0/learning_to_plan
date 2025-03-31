; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 622605

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child398 - child
    bread412 - bread-portion
    content255 - content-portion
    tray58 tray253 tray201 - tray
    table210 table348 table443 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray58 kitchen)
     (at tray253 kitchen)
     (at tray201 kitchen)
     (at_kitchen_bread bread412)
     (at_kitchen_content content255)
     (not_allergic_gluten child398)
     (waiting child398 table348)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child398)
    )
  )
)
