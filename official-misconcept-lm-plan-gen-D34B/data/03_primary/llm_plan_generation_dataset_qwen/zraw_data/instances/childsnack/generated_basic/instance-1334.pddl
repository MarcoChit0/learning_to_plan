; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 306849

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child116 child294 - child
    bread383 bread330 - bread-portion
    content57 content125 - content-portion
    tray13 tray346 tray8 - tray
    table407 table39 table357 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray13 kitchen)
     (at tray346 kitchen)
     (at tray8 kitchen)
     (at_kitchen_bread bread383)
     (at_kitchen_bread bread330)
     (at_kitchen_content content57)
     (at_kitchen_content content125)
     (not_allergic_gluten child116)
     (not_allergic_gluten child294)
     (waiting child116 table357)
     (waiting child294 table357)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child116)
     (served child294)
    )
  )
)
