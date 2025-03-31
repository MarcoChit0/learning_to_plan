; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 813736

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child313 - child
    bread195 - bread-portion
    content64 - content-portion
    tray144 tray423 tray457 tray398 - tray
    table102 table416 table197 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray144 kitchen)
     (at tray423 kitchen)
     (at tray457 kitchen)
     (at tray398 kitchen)
     (at_kitchen_bread bread195)
     (at_kitchen_content content64)
     (not_allergic_gluten child313)
     (waiting child313 table416)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child313)
    )
  )
)
