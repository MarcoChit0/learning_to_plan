; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 414123

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child399 child183 - child
    bread87 bread334 - bread-portion
    content144 content254 - content-portion
    tray233 tray25 - tray
    table462 table466 table331 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray233 kitchen)
     (at tray25 kitchen)
     (at_kitchen_bread bread87)
     (at_kitchen_bread bread334)
     (at_kitchen_content content144)
     (at_kitchen_content content254)
     (no_gluten_bread bread87)
     (no_gluten_content content254)
     (allergic_gluten child399)
     (not_allergic_gluten child183)
     (waiting child399 table462)
     (waiting child183 table331)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child399)
     (served child183)
    )
  )
)
