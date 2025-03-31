; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 569122

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child353 - child
    bread491 - bread-portion
    content335 - content-portion
    tray146 tray260 tray348 tray308 - tray
    table390 table86 table152 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray146 kitchen)
     (at tray260 kitchen)
     (at tray348 kitchen)
     (at tray308 kitchen)
     (at_kitchen_bread bread491)
     (at_kitchen_content content335)
     (not_allergic_gluten child353)
     (waiting child353 table152)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child353)
    )
  )
)
