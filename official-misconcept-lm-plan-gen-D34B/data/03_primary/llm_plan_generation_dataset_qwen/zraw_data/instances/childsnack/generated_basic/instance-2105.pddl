; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 781654

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child355 - child
    bread274 - bread-portion
    content145 - content-portion
    tray49 tray214 tray253 tray93 - tray
    table358 table99 table217 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray49 kitchen)
     (at tray214 kitchen)
     (at tray253 kitchen)
     (at tray93 kitchen)
     (at_kitchen_bread bread274)
     (at_kitchen_content content145)
     (not_allergic_gluten child355)
     (waiting child355 table358)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child355)
    )
  )
)
