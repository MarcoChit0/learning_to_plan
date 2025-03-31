; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 406701

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child414 child329 - child
    bread364 bread50 - bread-portion
    content229 content152 - content-portion
    tray332 - tray
    table141 table84 table440 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray332 kitchen)
     (at_kitchen_bread bread364)
     (at_kitchen_bread bread50)
     (at_kitchen_content content229)
     (at_kitchen_content content152)
     (not_allergic_gluten child329)
     (not_allergic_gluten child414)
     (waiting child414 table141)
     (waiting child329 table141)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child414)
     (served child329)
    )
  )
)
