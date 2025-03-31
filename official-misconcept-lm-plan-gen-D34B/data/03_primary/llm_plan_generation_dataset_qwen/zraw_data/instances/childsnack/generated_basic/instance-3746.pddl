; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 932462

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child32 - child
    bread488 - bread-portion
    content327 - content-portion
    tray297 tray410 tray438 - tray
    table477 table141 table321 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray297 kitchen)
     (at tray410 kitchen)
     (at tray438 kitchen)
     (at_kitchen_bread bread488)
     (at_kitchen_content content327)
     (not_allergic_gluten child32)
     (waiting child32 table141)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child32)
    )
  )
)
