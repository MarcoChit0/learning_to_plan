; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 845878

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child117 - child
    bread352 - bread-portion
    content430 - content-portion
    tray424 tray434 tray54 tray220 - tray
    table69 table64 table259 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray424 kitchen)
     (at tray434 kitchen)
     (at tray54 kitchen)
     (at tray220 kitchen)
     (at_kitchen_bread bread352)
     (at_kitchen_content content430)
     (not_allergic_gluten child117)
     (waiting child117 table69)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child117)
    )
  )
)
