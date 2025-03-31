; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 177907

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child263 child463 child30 - child
    bread120 bread224 bread337 - bread-portion
    content352 content473 content461 - content-portion
    tray99 tray390 tray20 - tray
    table388 table81 table136 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray99 kitchen)
     (at tray390 kitchen)
     (at tray20 kitchen)
     (at_kitchen_bread bread120)
     (at_kitchen_bread bread224)
     (at_kitchen_bread bread337)
     (at_kitchen_content content352)
     (at_kitchen_content content473)
     (at_kitchen_content content461)
     (not_allergic_gluten child263)
     (not_allergic_gluten child463)
     (not_allergic_gluten child30)
     (waiting child263 table81)
     (waiting child463 table81)
     (waiting child30 table136)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child263)
     (served child463)
     (served child30)
    )
  )
)
