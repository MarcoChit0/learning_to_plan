; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 427370

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child372 - child
    bread400 - bread-portion
    content30 - content-portion
    tray428 tray54 tray442 - tray
    table23 table173 table410 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray428 kitchen)
     (at tray54 kitchen)
     (at tray442 kitchen)
     (at_kitchen_bread bread400)
     (at_kitchen_content content30)
     (not_allergic_gluten child372)
     (waiting child372 table23)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child372)
    )
  )
)
