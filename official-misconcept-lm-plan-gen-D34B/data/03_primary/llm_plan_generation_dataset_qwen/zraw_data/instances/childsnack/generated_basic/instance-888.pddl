; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 840544

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child375 child304 child76 - child
    bread10 bread273 bread479 - bread-portion
    content229 content145 content142 - content-portion
    tray162 - tray
    table405 table50 table301 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray162 kitchen)
     (at_kitchen_bread bread10)
     (at_kitchen_bread bread273)
     (at_kitchen_bread bread479)
     (at_kitchen_content content229)
     (at_kitchen_content content145)
     (at_kitchen_content content142)
     (not_allergic_gluten child304)
     (not_allergic_gluten child76)
     (not_allergic_gluten child375)
     (waiting child375 table50)
     (waiting child304 table301)
     (waiting child76 table405)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child375)
     (served child304)
     (served child76)
    )
  )
)
