; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 973598

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child416 child463 - child
    bread393 bread269 - bread-portion
    content157 content338 - content-portion
    tray31 tray235 tray365 - tray
    table31 table232 table7 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray31 kitchen)
     (at tray235 kitchen)
     (at tray365 kitchen)
     (at_kitchen_bread bread393)
     (at_kitchen_bread bread269)
     (at_kitchen_content content157)
     (at_kitchen_content content338)
     (not_allergic_gluten child463)
     (not_allergic_gluten child416)
     (waiting child416 table232)
     (waiting child463 table31)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child416)
     (served child463)
    )
  )
)
