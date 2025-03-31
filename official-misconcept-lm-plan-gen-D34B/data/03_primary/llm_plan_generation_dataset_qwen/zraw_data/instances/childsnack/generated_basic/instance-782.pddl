; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 613064

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child136 - child
    bread153 - bread-portion
    content8 - content-portion
    tray144 tray51 - tray
    table286 table346 table294 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray144 kitchen)
     (at tray51 kitchen)
     (at_kitchen_bread bread153)
     (at_kitchen_content content8)
     (not_allergic_gluten child136)
     (waiting child136 table286)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child136)
    )
  )
)
