; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 160483

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child144 child221 - child
    bread55 bread37 - bread-portion
    content252 content287 - content-portion
    tray17 tray382 - tray
    table405 table42 table388 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray17 kitchen)
     (at tray382 kitchen)
     (at_kitchen_bread bread55)
     (at_kitchen_bread bread37)
     (at_kitchen_content content252)
     (at_kitchen_content content287)
     (not_allergic_gluten child144)
     (not_allergic_gluten child221)
     (waiting child144 table388)
     (waiting child221 table42)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child144)
     (served child221)
    )
  )
)
