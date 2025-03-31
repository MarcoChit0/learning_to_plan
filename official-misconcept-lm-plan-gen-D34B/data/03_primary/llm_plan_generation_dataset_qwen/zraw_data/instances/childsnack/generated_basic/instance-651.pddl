; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.3
; random seed: 835568

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child226 child96 - child
    bread236 bread229 - bread-portion
    content497 content225 - content-portion
    tray482 tray248 - tray
    table8 table313 table303 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray482 kitchen)
     (at tray248 kitchen)
     (at_kitchen_bread bread236)
     (at_kitchen_bread bread229)
     (at_kitchen_content content497)
     (at_kitchen_content content225)
     (no_gluten_bread bread236)
     (no_gluten_content content225)
     (allergic_gluten child96)
     (not_allergic_gluten child226)
     (waiting child226 table8)
     (waiting child96 table303)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child226)
     (served child96)
    )
  )
)
