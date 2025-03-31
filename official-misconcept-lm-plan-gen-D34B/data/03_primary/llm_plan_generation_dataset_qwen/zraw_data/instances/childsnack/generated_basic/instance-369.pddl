; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 560977

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child363 - child
    bread64 - bread-portion
    content63 - content-portion
    tray44 tray285 tray380 tray313 - tray
    table232 table69 table331 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray44 kitchen)
     (at tray285 kitchen)
     (at tray380 kitchen)
     (at tray313 kitchen)
     (at_kitchen_bread bread64)
     (at_kitchen_content content63)
     (not_allergic_gluten child363)
     (waiting child363 table69)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child363)
    )
  )
)
