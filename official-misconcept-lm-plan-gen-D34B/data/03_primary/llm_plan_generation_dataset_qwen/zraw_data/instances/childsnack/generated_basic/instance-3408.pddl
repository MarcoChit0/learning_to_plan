; child-snack task with 3 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 843507

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child372 child409 child245 - child
    bread47 bread426 bread186 - bread-portion
    content478 content81 content296 - content-portion
    tray152 tray185 tray381 - tray
    table278 table481 table83 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray152 kitchen)
     (at tray185 kitchen)
     (at tray381 kitchen)
     (at_kitchen_bread bread47)
     (at_kitchen_bread bread426)
     (at_kitchen_bread bread186)
     (at_kitchen_content content478)
     (at_kitchen_content content81)
     (at_kitchen_content content296)
     (no_gluten_bread bread426)
     (no_gluten_content content296)
     (allergic_gluten child372)
     (not_allergic_gluten child409)
     (not_allergic_gluten child245)
     (waiting child372 table83)
     (waiting child409 table481)
     (waiting child245 table278)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child372)
     (served child409)
     (served child245)
    )
  )
)
