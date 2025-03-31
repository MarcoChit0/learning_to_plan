; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 3345

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child459 - child
    bread402 - bread-portion
    content22 - content-portion
    tray245 tray67 tray323 tray309 - tray
    table340 table453 table102 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray245 kitchen)
     (at tray67 kitchen)
     (at tray323 kitchen)
     (at tray309 kitchen)
     (at_kitchen_bread bread402)
     (at_kitchen_content content22)
     (not_allergic_gluten child459)
     (waiting child459 table102)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child459)
    )
  )
)
