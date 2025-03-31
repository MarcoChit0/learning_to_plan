; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 600765

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child141 - child
    bread442 - bread-portion
    content335 - content-portion
    tray436 tray414 tray125 tray127 - tray
    table96 table342 table18 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray436 kitchen)
     (at tray414 kitchen)
     (at tray125 kitchen)
     (at tray127 kitchen)
     (at_kitchen_bread bread442)
     (at_kitchen_content content335)
     (not_allergic_gluten child141)
     (waiting child141 table342)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child141)
    )
  )
)
