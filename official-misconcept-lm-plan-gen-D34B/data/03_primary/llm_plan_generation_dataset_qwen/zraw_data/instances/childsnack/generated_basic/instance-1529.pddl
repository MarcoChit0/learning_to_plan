; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 213895

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child384 child170 - child
    bread348 bread395 - bread-portion
    content90 content286 - content-portion
    tray242 tray272 tray224 tray212 - tray
    table462 table228 table26 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray242 kitchen)
     (at tray272 kitchen)
     (at tray224 kitchen)
     (at tray212 kitchen)
     (at_kitchen_bread bread348)
     (at_kitchen_bread bread395)
     (at_kitchen_content content90)
     (at_kitchen_content content286)
     (not_allergic_gluten child170)
     (not_allergic_gluten child384)
     (waiting child384 table26)
     (waiting child170 table26)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child384)
     (served child170)
    )
  )
)
