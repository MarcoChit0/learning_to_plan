; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 850233

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child388 - child
    bread479 - bread-portion
    content256 - content-portion
    tray382 tray466 tray190 - tray
    table365 table391 table411 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray382 kitchen)
     (at tray466 kitchen)
     (at tray190 kitchen)
     (at_kitchen_bread bread479)
     (at_kitchen_content content256)
     (not_allergic_gluten child388)
     (waiting child388 table391)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child388)
    )
  )
)
