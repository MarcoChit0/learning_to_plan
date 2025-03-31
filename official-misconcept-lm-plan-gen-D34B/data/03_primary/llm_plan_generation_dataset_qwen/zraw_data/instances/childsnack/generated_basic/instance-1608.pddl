; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 68597

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child334 - child
    bread38 - bread-portion
    content202 - content-portion
    tray425 tray88 tray305 tray153 - tray
    table353 table200 table115 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray425 kitchen)
     (at tray88 kitchen)
     (at tray305 kitchen)
     (at tray153 kitchen)
     (at_kitchen_bread bread38)
     (at_kitchen_content content202)
     (not_allergic_gluten child334)
     (waiting child334 table115)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child334)
    )
  )
)
