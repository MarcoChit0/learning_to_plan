; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 510410

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child453 child208 - child
    bread336 bread412 - bread-portion
    content207 content78 - content-portion
    tray268 - tray
    table121 table18 table179 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray268 kitchen)
     (at_kitchen_bread bread336)
     (at_kitchen_bread bread412)
     (at_kitchen_content content207)
     (at_kitchen_content content78)
     (no_gluten_bread bread412)
     (no_gluten_content content78)
     (allergic_gluten child453)
     (not_allergic_gluten child208)
     (waiting child453 table121)
     (waiting child208 table18)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child453)
     (served child208)
    )
  )
)
