; child-snack task with 3 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 664837

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child462 child189 child30 - child
    bread445 bread221 bread233 - bread-portion
    content357 content61 content433 - content-portion
    tray386 tray145 tray368 tray266 - tray
    table151 table463 table413 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray386 kitchen)
     (at tray145 kitchen)
     (at tray368 kitchen)
     (at tray266 kitchen)
     (at_kitchen_bread bread445)
     (at_kitchen_bread bread221)
     (at_kitchen_bread bread233)
     (at_kitchen_content content357)
     (at_kitchen_content content61)
     (at_kitchen_content content433)
     (no_gluten_bread bread221)
     (no_gluten_content content357)
     (allergic_gluten child30)
     (not_allergic_gluten child189)
     (not_allergic_gluten child462)
     (waiting child462 table463)
     (waiting child189 table413)
     (waiting child30 table413)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child462)
     (served child189)
     (served child30)
    )
  )
)
