; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 685856

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child97 child73 - child
    bread77 bread24 - bread-portion
    content29 content310 - content-portion
    tray19 - tray
    table422 table456 table227 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray19 kitchen)
     (at_kitchen_bread bread77)
     (at_kitchen_bread bread24)
     (at_kitchen_content content29)
     (at_kitchen_content content310)
     (no_gluten_bread bread77)
     (no_gluten_content content29)
     (allergic_gluten child97)
     (not_allergic_gluten child73)
     (waiting child97 table227)
     (waiting child73 table456)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child97)
     (served child73)
    )
  )
)
