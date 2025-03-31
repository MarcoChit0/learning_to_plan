; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 760240

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child332 - child
    bread114 - bread-portion
    content259 - content-portion
    tray285 tray165 tray421 tray127 - tray
    table233 table298 table416 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray285 kitchen)
     (at tray165 kitchen)
     (at tray421 kitchen)
     (at tray127 kitchen)
     (at_kitchen_bread bread114)
     (at_kitchen_content content259)
     (not_allergic_gluten child332)
     (waiting child332 table416)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child332)
    )
  )
)
