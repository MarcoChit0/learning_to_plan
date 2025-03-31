; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 428590

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child130 child61 - child
    bread401 bread333 - bread-portion
    content58 content181 - content-portion
    tray493 tray262 - tray
    table267 table229 table89 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray493 kitchen)
     (at tray262 kitchen)
     (at_kitchen_bread bread401)
     (at_kitchen_bread bread333)
     (at_kitchen_content content58)
     (at_kitchen_content content181)
     (not_allergic_gluten child130)
     (not_allergic_gluten child61)
     (waiting child130 table89)
     (waiting child61 table267)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child130)
     (served child61)
    )
  )
)
