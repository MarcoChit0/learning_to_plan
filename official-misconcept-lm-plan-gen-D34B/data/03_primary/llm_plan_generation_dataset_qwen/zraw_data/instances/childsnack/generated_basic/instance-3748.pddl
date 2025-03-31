; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 574484

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child143 - child
    bread156 - bread-portion
    content317 - content-portion
    tray32 tray353 tray419 - tray
    table79 table207 table153 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray32 kitchen)
     (at tray353 kitchen)
     (at tray419 kitchen)
     (at_kitchen_bread bread156)
     (at_kitchen_content content317)
     (not_allergic_gluten child143)
     (waiting child143 table207)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child143)
    )
  )
)
