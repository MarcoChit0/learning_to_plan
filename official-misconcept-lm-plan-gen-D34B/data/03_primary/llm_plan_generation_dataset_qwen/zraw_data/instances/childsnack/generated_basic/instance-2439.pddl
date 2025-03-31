; child-snack task with 3 children and 0.59 gluten factor 
; constant factor of 1.3
; random seed: 726366

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child456 child23 child155 - child
    bread110 bread16 bread419 - bread-portion
    content318 content387 content64 - content-portion
    tray475 - tray
    table186 table302 table378 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray475 kitchen)
     (at_kitchen_bread bread110)
     (at_kitchen_bread bread16)
     (at_kitchen_bread bread419)
     (at_kitchen_content content318)
     (at_kitchen_content content387)
     (at_kitchen_content content64)
     (no_gluten_bread bread110)
     (no_gluten_content content318)
     (allergic_gluten child155)
     (not_allergic_gluten child456)
     (not_allergic_gluten child23)
     (waiting child456 table302)
     (waiting child23 table378)
     (waiting child155 table186)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child456)
     (served child23)
     (served child155)
    )
  )
)
