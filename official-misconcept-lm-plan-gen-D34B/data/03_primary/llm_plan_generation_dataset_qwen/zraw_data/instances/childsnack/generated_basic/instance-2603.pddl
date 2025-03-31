; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 492318

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child277 - child
    bread172 - bread-portion
    content324 - content-portion
    tray381 tray437 tray238 - tray
    table166 table40 table168 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray381 kitchen)
     (at tray437 kitchen)
     (at tray238 kitchen)
     (at_kitchen_bread bread172)
     (at_kitchen_content content324)
     (not_allergic_gluten child277)
     (waiting child277 table40)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child277)
    )
  )
)
