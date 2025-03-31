; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 795677

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child323 child416 - child
    bread93 bread288 - bread-portion
    content388 content164 - content-portion
    tray168 - tray
    table129 table183 table50 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray168 kitchen)
     (at_kitchen_bread bread93)
     (at_kitchen_bread bread288)
     (at_kitchen_content content388)
     (at_kitchen_content content164)
     (not_allergic_gluten child416)
     (not_allergic_gluten child323)
     (waiting child323 table129)
     (waiting child416 table183)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child323)
     (served child416)
    )
  )
)
