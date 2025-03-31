; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 916294

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child281 child3 - child
    bread167 bread324 - bread-portion
    content372 content299 - content-portion
    tray401 tray404 tray439 tray432 - tray
    table490 table339 table260 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray401 kitchen)
     (at tray404 kitchen)
     (at tray439 kitchen)
     (at tray432 kitchen)
     (at_kitchen_bread bread167)
     (at_kitchen_bread bread324)
     (at_kitchen_content content372)
     (at_kitchen_content content299)
     (not_allergic_gluten child281)
     (not_allergic_gluten child3)
     (waiting child281 table260)
     (waiting child3 table339)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child281)
     (served child3)
    )
  )
)
