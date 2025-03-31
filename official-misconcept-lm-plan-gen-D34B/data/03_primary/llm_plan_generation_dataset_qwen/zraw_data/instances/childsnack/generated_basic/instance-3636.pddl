; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 75761

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child334 child89 - child
    bread298 bread92 - bread-portion
    content201 content387 - content-portion
    tray381 tray229 tray128 tray51 - tray
    table463 table364 table372 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray381 kitchen)
     (at tray229 kitchen)
     (at tray128 kitchen)
     (at tray51 kitchen)
     (at_kitchen_bread bread298)
     (at_kitchen_bread bread92)
     (at_kitchen_content content201)
     (at_kitchen_content content387)
     (no_gluten_bread bread298)
     (no_gluten_content content201)
     (allergic_gluten child334)
     (not_allergic_gluten child89)
     (waiting child334 table463)
     (waiting child89 table372)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child334)
     (served child89)
    )
  )
)
