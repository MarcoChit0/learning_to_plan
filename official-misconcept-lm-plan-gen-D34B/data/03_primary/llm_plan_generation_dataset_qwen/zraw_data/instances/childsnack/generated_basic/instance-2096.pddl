; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 232201

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child495 - child
    bread489 - bread-portion
    content290 - content-portion
    tray358 tray19 tray28 tray237 - tray
    table391 table274 table114 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray358 kitchen)
     (at tray19 kitchen)
     (at tray28 kitchen)
     (at tray237 kitchen)
     (at_kitchen_bread bread489)
     (at_kitchen_content content290)
     (not_allergic_gluten child495)
     (waiting child495 table391)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child495)
    )
  )
)
