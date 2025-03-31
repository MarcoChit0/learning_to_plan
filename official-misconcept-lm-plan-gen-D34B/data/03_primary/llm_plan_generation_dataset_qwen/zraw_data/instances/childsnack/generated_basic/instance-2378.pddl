; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 333297

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child270 child338 - child
    bread486 bread478 - bread-portion
    content177 content78 - content-portion
    tray29 - tray
    table266 table252 table292 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray29 kitchen)
     (at_kitchen_bread bread486)
     (at_kitchen_bread bread478)
     (at_kitchen_content content177)
     (at_kitchen_content content78)
     (not_allergic_gluten child338)
     (not_allergic_gluten child270)
     (waiting child270 table266)
     (waiting child338 table292)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child270)
     (served child338)
    )
  )
)
