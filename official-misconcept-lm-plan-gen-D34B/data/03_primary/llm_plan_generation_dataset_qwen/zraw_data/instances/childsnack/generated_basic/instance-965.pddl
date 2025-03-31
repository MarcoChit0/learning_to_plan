; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 2284

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child200 child387 child356 - child
    bread190 bread272 bread6 - bread-portion
    content77 content332 content81 - content-portion
    tray119 tray141 tray341 tray473 - tray
    table29 table476 table315 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray119 kitchen)
     (at tray141 kitchen)
     (at tray341 kitchen)
     (at tray473 kitchen)
     (at_kitchen_bread bread190)
     (at_kitchen_bread bread272)
     (at_kitchen_bread bread6)
     (at_kitchen_content content77)
     (at_kitchen_content content332)
     (at_kitchen_content content81)
     (not_allergic_gluten child356)
     (not_allergic_gluten child200)
     (not_allergic_gluten child387)
     (waiting child200 table476)
     (waiting child387 table476)
     (waiting child356 table476)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child200)
     (served child387)
     (served child356)
    )
  )
)
