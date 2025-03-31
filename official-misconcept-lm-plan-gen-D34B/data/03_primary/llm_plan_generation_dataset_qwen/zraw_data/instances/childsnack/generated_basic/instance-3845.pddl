; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 799339

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child253 - child
    bread486 - bread-portion
    content436 - content-portion
    tray149 tray239 tray55 tray33 - tray
    table181 table324 table301 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray149 kitchen)
     (at tray239 kitchen)
     (at tray55 kitchen)
     (at tray33 kitchen)
     (at_kitchen_bread bread486)
     (at_kitchen_content content436)
     (not_allergic_gluten child253)
     (waiting child253 table181)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child253)
    )
  )
)
