; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 662629

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child313 - child
    bread246 - bread-portion
    content319 - content-portion
    tray338 tray403 tray172 tray420 - tray
    table294 table392 table455 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray338 kitchen)
     (at tray403 kitchen)
     (at tray172 kitchen)
     (at tray420 kitchen)
     (at_kitchen_bread bread246)
     (at_kitchen_content content319)
     (not_allergic_gluten child313)
     (waiting child313 table455)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child313)
    )
  )
)
