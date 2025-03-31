; child-snack task with 3 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 698760

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child353 child442 child399 - child
    bread480 bread363 bread45 - bread-portion
    content331 content437 content362 - content-portion
    tray248 tray230 tray387 - tray
    table380 table102 table210 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray248 kitchen)
     (at tray230 kitchen)
     (at tray387 kitchen)
     (at_kitchen_bread bread480)
     (at_kitchen_bread bread363)
     (at_kitchen_bread bread45)
     (at_kitchen_content content331)
     (at_kitchen_content content437)
     (at_kitchen_content content362)
     (no_gluten_bread bread363)
     (no_gluten_content content331)
     (allergic_gluten child399)
     (not_allergic_gluten child442)
     (not_allergic_gluten child353)
     (waiting child353 table380)
     (waiting child442 table380)
     (waiting child399 table102)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child353)
     (served child442)
     (served child399)
    )
  )
)
