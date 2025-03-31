; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 318818

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child432 - child
    bread286 - bread-portion
    content381 - content-portion
    tray87 tray267 tray11 - tray
    table335 table329 table84 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray87 kitchen)
     (at tray267 kitchen)
     (at tray11 kitchen)
     (at_kitchen_bread bread286)
     (at_kitchen_content content381)
     (not_allergic_gluten child432)
     (waiting child432 table329)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child432)
    )
  )
)
