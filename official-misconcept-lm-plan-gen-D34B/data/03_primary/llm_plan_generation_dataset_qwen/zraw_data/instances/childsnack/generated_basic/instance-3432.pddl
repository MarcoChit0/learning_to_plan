; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 538178

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child95 - child
    bread80 - bread-portion
    content395 - content-portion
    tray472 tray323 tray168 tray359 - tray
    table153 table463 table132 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray472 kitchen)
     (at tray323 kitchen)
     (at tray168 kitchen)
     (at tray359 kitchen)
     (at_kitchen_bread bread80)
     (at_kitchen_content content395)
     (not_allergic_gluten child95)
     (waiting child95 table153)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child95)
    )
  )
)
