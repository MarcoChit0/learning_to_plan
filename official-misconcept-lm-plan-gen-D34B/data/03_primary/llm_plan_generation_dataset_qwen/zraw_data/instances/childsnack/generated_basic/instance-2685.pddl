; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 553199

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child174 child129 child355 - child
    bread445 bread243 bread339 - bread-portion
    content20 content396 content146 - content-portion
    tray311 tray270 tray268 - tray
    table449 table409 table334 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray311 kitchen)
     (at tray270 kitchen)
     (at tray268 kitchen)
     (at_kitchen_bread bread445)
     (at_kitchen_bread bread243)
     (at_kitchen_bread bread339)
     (at_kitchen_content content20)
     (at_kitchen_content content396)
     (at_kitchen_content content146)
     (not_allergic_gluten child129)
     (not_allergic_gluten child174)
     (not_allergic_gluten child355)
     (waiting child174 table409)
     (waiting child129 table449)
     (waiting child355 table334)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child174)
     (served child129)
     (served child355)
    )
  )
)
